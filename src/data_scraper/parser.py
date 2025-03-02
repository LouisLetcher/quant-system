class Parser:
    @staticmethod
    def parse_data(data):
        """Formats data for consistency before storage."""
        data.columns = [col.lower().replace(" ", "_") for col in data.columns]
        return data