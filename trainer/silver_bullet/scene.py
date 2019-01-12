from pybullet_utils import bullet_client

class Scene(object):
    def __init__(self, gravity, timestep, frame_skip, client=None):
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.dt = timestep * frame_skip

        self.connect(client)
        self.episode_restart()

    def connect(self, client):
        if client:
            self.client = client
        else:
            self.client = bullet_client.BulletClient()

    def clean_everything(self):
        self.client.resetSimulation()
        self.configure_simulation()

    def configure_simulation(self):
        self.client.setGravity(0, 0, -self.gravity)
        self.client.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSubSteps=self.frame_skip)

    def load_plane(self):
        self.plane_id = self.client.loadURDF("plane.urdf")

    def episode_restart(self):
        self.clean_everything()
        self.load_plane()

    def step(self):
        self.client.stepSimulation()
