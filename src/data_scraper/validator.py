class Validator:
    @staticmethod
    def validate_data(data):
        """Checks if the data is valid and non-empty."""
        if data is None or data.empty:
            raise ValueError("Invalid or empty data received.")
        return True