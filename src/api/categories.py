from flask_restplus import Resource, Api, request
from categories_utils import (get_all_categories, create_category,update_category,delete_category)

api = Api(app)


ns = api.namespace('machinelearning/categories', description='Options related to Machine Learning categories')

@ns.route('/')
class CategoryCollection(Resource):
    def get(self):
        """Return a list of machine learning categories"""
        return get_all_categories()

    @api.response(201, 'Category succesfully created')
    def post(self):
        """Creates a new machine learning category"""
        create_category(request.json)
        return None, 201

@ns.route('/<int:id>')
@api.response(404, 'Category not found')
class CategoryItem(Resource):

    def get(self, id):
        """Returns details of a Machine Learning Category"""
        return get_category(id)

    @api.response(2-4, 'Category succesfully updated.')
    def put(self, id):
        """Updates Machine Learning Category"""
        update_category(id, request.json)
        return None, 204

    @api.response(204, 'Machine Learning Category succesfully deleted.')
    def delete(self,id):
        """Deletes a Machine Learning category."""
        delete_category(id)
        return None, 204