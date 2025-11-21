from pythreejs import *
from ipywidgets import embed
import os

class Viewer:
    def __init__(self,):
        self.scene = Scene(background="#dddddd")
        self.camera = PerspectiveCamera(
            position = [3, 3, 3],
            up = [0, 1, 0],
            children=[DirectionalLight(color='white', position=[3,5,1], intensity=0.5)]
        )

        ambient_light = AmbientLight(color='#777777')
        self.scene.add(ambient_light)

        self.controls = OrbitControls(controlling=self.camera)
        self.renderer = Renderer(
            camera=self.camera, 
            scene=self.scene, 
            controls=[self.controls], 
            width=600, height=600)
        
        # Example Geometry
        geometry = BoxGeometry(width=1, height=1, depth=1)
        material = MeshStandardMaterial(color='skyblue', roughness=0.5)
        cube = Mesh(geometry=geometry, material=material)

        self.scene.add(cube)
        
        return
    
    def to_html(
        self, 
        save_path= "/root/project/results/view.html"
    )-> None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok= True)
        embed.embed_minimal_html(save_path , views=[self.renderer])     
        print(f"Saved to {save_path}")
        return