

class AppLauncher:
    """Minimal fallback that mimics Orbit's AppLauncher enough to start Kit."""
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true", help="Run without UI")
        return parser

    def __init__(self, args):
        from omni.isaac.kit import SimulationApp
        self.args = args

    def launch(self):
        # Use Isaac Sim's Kit entrypoint if available in the environment
        from omni.isaac.kit import SimulationApp
        return SimulationApp({"headless": bool(getattr(self.args, "headless", False))})