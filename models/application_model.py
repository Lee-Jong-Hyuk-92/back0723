from pymongo.collection import Collection

class ApplicationModel:
    def __init__(self, collection: Collection):
        self.collection = collection

    def create_application(self, inference_result_id, status="신청"):
        doc = {
            "inference_result_id": inference_result_id,
            "status": status
        }
        return self.collection.insert_one(doc)