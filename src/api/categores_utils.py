import requests
from minio import Minio
from minio.error import S3Error
import jsonify
import json

def create_client():
    return client = Minio(
        "play.min.io",
        access_key="",
        secret_key="",
    )

def get_all_categories():
    client = create_client()
    return jsonify(client.list_buckets)

def create_category(jsonObj):
    client = create_client()
    client.make_bucket(json.load(jsonObj))

def update_category(id, jsonObj):
    client = create_client()
    client.remove_bucket(id)
    client.make_bucket(jsonObj)


def delete_category(id):
    client = create_client()
    if client.bucket_exists(id):
        client.remove_bucket(id)

def main():
    return None

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)
