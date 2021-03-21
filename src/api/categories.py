from flask_restplus import Reasource, Api

api = Api(app)


ns = api.namespace('machinelearning/categories', description='Options related to Machine Learning categories')

@ns.route('/')
class CategoryCollection(Reasource):
    def get(self):
        """Return a list of machine learning categories"""
        return get_all_categories()

    @api.response(201, 'Category succesfully created')
    def post(self):
        """Creates a new machine learning category"""