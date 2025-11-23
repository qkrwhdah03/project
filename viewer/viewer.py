import viser
from utils import wait

class Viewer:
    def __init__(self):
        self.server = viser.ViserServer(host="0.0.0.0", port=8080)
        self.server.scene.set_environment_map(background=True)

        self.server.scene.add_box(
            name = "box"
        )

    def run(self):
        print("Viewer running at http://localhost:8080")
        wait()


if __name__ == "__main__":
    viewer = Viewer()
    viewer.run()