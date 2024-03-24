class PredictionFilter:
    def __init__(self, target: list[str]) -> None:
        self.target = target
        self.prediction_result = []

    def add_prediction(self, predictions: list, gps_coordinate: tuple) -> None:
        """Add prediction result
        Args:
            predictions: list of prediction in form of [{"class":"classname", "conf": confidence_score}]
            gps_coordinate: gps coordinate (lat, long)
        """
        self.prediction_result.append(predictions)

    def get_result() -> dict:
        """Return the most confidence prediction result that matches the target as dictionary with {"target": (lat , long)}"""
        #TODO: figure out logic for filtering repeated detection result, find closest shape in case we can't find the exact one
        pass
