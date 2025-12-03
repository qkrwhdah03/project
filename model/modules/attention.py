# model/modules/attention.py
# Attention modules for CubeDiff (Stable Diffusion 2)

# NOTE: Cited from https://github.com/Juan5713/OpenCubeDiff/ (Open source)

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from einops import rearrange
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.utils import deprecate
import torch.nn.functional as F

class CubeDiffAttnProcessor:
    """
    A custom processor for CubeDiff that uses PyTorch 2.0+ scaled dot-product attention
    without modifying or reshaping the attention mask.
    """
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CubeDiffAttnProcessor requires PyTorch 2.0+. Please upgrade PyTorch to use this.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads

        # [B, H, L, D]
        # [B L D] -> [B, L, H, D] -> [B, H, L, D]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class CubeDiffTransformerBlock(BasicTransformerBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_faces = 6
        self.attn1.set_processor(CubeDiffAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # Fixed: Add attention_mask parameter
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention

        bt, hw, _ = hidden_states.shape

        T = self.num_faces
        B = bt // T

        # Normalization layer; by default should be layer norm on the hidden states 
        # which is the case for stable diffusion we are adapting, but we leave the if-else for flexibility and keeping the original code intact
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(bt, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

  
        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # reshape to attend to all faces
        norm_hidden_states = rearrange(norm_hidden_states, "(b t) (hw) c -> b (t hw) c", b=B, t=T, hw=hw)

        # Assume that the drop face is the front face
        '''
        front_face_drop = cross_attention_kwargs.pop("front_face_drop", False)

        if front_face_drop:
            # Right now it's a bit hacky, because we drop front face 
            # For the whole minibatch with probability 10%, as opposed to 
            # Dropping the front face for each sample independently. This is because
            # Using the mask would cause the backend to always use math mode instead of flashattention, which is much slower.
            with torch.no_grad():
                # [B, H, Q, K]
                # This should work.... Since it broadcasts
                self_attention_mask = torch.ones((1, 1, 1, T*hw), dtype=torch.bool, device=hidden_states.device)
                self_attention_mask[:, :, :, :hw] = False
        else:
            self_attention_mask = None
        '''

        drop_face_index = cross_attention_kwargs.pop("drop_face_index", None)

        if drop_face_index is not None:
            with torch.no_grad():
                self_attention_mask = torch.ones((1, 1, 1, T*hw), dtype=torch.bool, device=hidden_states.device)
                start_index = drop_face_index * hw
                end_index = (drop_face_index + 1) * hw
                self_attention_mask[:, :, :, start_index:end_index] = False
        else:
            self_attention_mask = None

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=self_attention_mask,
            **cross_attention_kwargs,
        )

        # Delete the attention mask to save memory
        del self_attention_mask
 
        # reshape back to (B*T, C, H, W) post attention
        attn_output = rearrange(attn_output, "b (t hw) c -> (b t) (hw) c", b=B, t=T, hw=hw)

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # -------- SMALL MODIFICATION AS WE DO NOT HAVE THE CHUNK FUNCTION ----------

        ff_output = self.ff(norm_hidden_states)

        # -------- END OF MODIFICATION ----------

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

def swap_transformer_blocks(root: nn.Module) -> None:
    """
    Replace every `BasicTransformerBlock` inside `Transformer2DModel`
    """
    for child in root.children():
        swap_transformer_blocks(child)
        if isinstance(child, Transformer2DModel):
            for i, blk in enumerate(child.transformer_blocks):
                if isinstance(blk, BasicTransformerBlock):
                    new_blk = CubeDiffTransformerBlock(
                        dim=blk.dim,
                        num_attention_heads=blk.num_attention_heads,
                        attention_head_dim=blk.attention_head_dim,
                        dropout=blk.dropout,
                        cross_attention_dim=blk.cross_attention_dim,
                        activation_fn=blk.activation_fn,
                        num_embeds_ada_norm=getattr(child.config, 'num_embeds_ada_norm', None),
                        attention_bias=blk.attention_bias,
                        only_cross_attention=blk.only_cross_attention,
                        double_self_attention=blk.double_self_attention,
                        norm_elementwise_affine=blk.norm_elementwise_affine,
                        norm_type=blk.norm_type,
                        norm_eps=getattr(child.config, 'norm_eps', 1e-5),
                        upcast_attention=getattr(child.config, 'upcast_attention', False),
                        attention_type=getattr(child.config, 'attention_type', 'default'),
                    )
                    # Load the state dict with proper error handling
                    try:
                        new_blk.load_state_dict(blk.state_dict(), strict=False)
                    except RuntimeError as e:
                        print(f"Warning: Could not load state dict completely: {e}")
                        # Copy compatible weights manually
                        new_state = new_blk.state_dict()
                        old_state = blk.state_dict()
                        for key in new_state.keys():
                            if key in old_state and new_state[key].shape == old_state[key].shape:
                                new_state[key].copy_(old_state[key])
                        new_blk.load_state_dict(new_state)
                    
                    child.transformer_blocks[i] = new_blk
