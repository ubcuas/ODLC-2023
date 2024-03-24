class PredictionFilter:
    def __init__(self, target: list[str]) -> None:
        self.target = target

    def add_prediction(predictions: list, gps_coordinate: tuple) -> None:
        """Add prediction result with 
        Args:
            predictions: list of prediction in form of [{"class":"classname", "conf": confidence_score}]
            gps_coordinate: gps coordinate (lat, long)
        """
        pass

    def get_result() -> dict:
        """Return the most confidence prediction result that matches the target as dictionary with {"target": (lat , long)}"""
        #TODO: figure out logic for filtering repeated detection result, find closest shape in case we can't find the exact one
        pass
