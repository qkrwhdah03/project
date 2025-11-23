from viewer import CubeMapViewer, CubeMap


if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser(description="Convert equirectangular StreetView images to cubemap faces.")
    parser.add_argument("--cube_x", type=int, default= 100,
                        help="Size of rendered cubemap in x-axis")
    parser.add_argument("--cube_y", type=int, default= 100,
                    help="Size of rendered cubemap in y-axis")
    parser.add_argument("--cube_z", type=int, default= 100,
                    help="Size of rendered cubemap in z-axis")
    parser.add_argument("--name", type=str, default="cubemap",
                        help="Name of cubemap")
    parser.add_argument("--dir_path", type=str, default="/root/project/data/cubemap",
                        help="Cubemap data directory path")
    parser.add_argument("--prefix", type=str, help="Cubemap data prefix")
    parser.add_argument("--ext", type=str, default=".png",
                        help="Cubemap data file extension")
   
    
    args = parser.parse_args()

    viewer = CubeMapViewer(cube_render_x= args.cube_x, cube_render_y= args.cube_y, cube_render_z= args.cube_z)
    cubemap = CubeMap(name= args.name, dir_path= args.dir_path, prefix= args.prefix, ext = args.ext)
    viewer.add_cubemap(cubemap)
    viewer.run()