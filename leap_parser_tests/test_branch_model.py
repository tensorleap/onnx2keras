import unittest
from pathlib import Path
import sys
print(sys.path)
from leap_model_parser.contract.graph import Node, ConnectionOutput, OutputData, ConnectionInput, \
    InputData, WrapperData
from leap_model_parser.contract.importmodelresponse import ImportModelTypeEnum
from leap_model_parser.model_parser import ModelParser
import pytest
import boto3
import os
import shutil
import tempfile
import argparse

BUCKET_NAME = 'tensorleap-engine-tests-dev'
PREFIX = 'onnx2keras'
if not 'LOCAL_TEST' in os.environ:
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        aws_session_token=os.environ['AWS_SESSION_TOKEN'],
        region_name='us-east-1'
    )

    def download_from_s3(aws_dir, dest_dir="", is_temp=False):
        if 'LOCAL_TEST' in os.environ:
            return dest_dir, is_temp
        real_dir = ""
        if not is_temp:
            real_dir = dest_dir
            if len(dest_dir) == 0:
                raise Exception("Need to provide destination dir if non-temp directory is used for file downloading")
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
        else:
            # Create a temporary directory
            real_dir = tempfile.mkdtemp()
        path = f"{PREFIX}/{aws_dir}"  # Use the provided directory as the prefix
        # List objects in the bucket with the specified prefix
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=path
        )

        # Download files to the temporary directory
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if obj['Size'] > 0:
                    rel_path = key[len(path):].lstrip("/")
                    dirname = os.path.dirname(rel_path)
                    full_dir = os.path.join(real_dir, dirname)
                    if len(full_dir) > 0 and not os.path.exists(full_dir):
                        os.makedirs(full_dir)
                    filename = os.path.join(real_dir, rel_path)
                    if not os.path.exists(filename):
                        s3.download_file(BUCKET_NAME, key, filename)
                        print(f"Downloaded {key} to {filename}")
        else:
            print("No objects found under the specified prefix.")

        # Provide the temporary directory path to the test function
        return real_dir


def test_branch_model(cloud_dir, model_name):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cloud_dir", type=str, required=True)
    # parser.add_argument("--model_name", type=str, required=True)
    # args = parser.parse_args()
    cloud_dir = cloud_dir + "/"
    # model_name = args.model_name

    #TODO fix download only file - not directory
    print("Downloading Assets...")
    real_dir = download_from_s3(cloud_dir, cloud_dir)
    model_path = Path(real_dir) / model_name
    suffix = model_name.split(".")[-1]
    if not os.path.exists(model_path):
        raise Exception(f"Model {model_path} does not exist")
    if suffix == "onnx":
        graph, connected_inputs = ModelParser(
            should_transform_inputs_and_outputs=True).generate_model_graph(model_path, ImportModelTypeEnum.ONNX)
    elif suffix == "h5":
        graph, connected_inputs = ModelParser().generate_model_graph(
            model_path, ImportModelTypeEnum.H5_TF2)
    else:
        raise Exception(f"The provided model path {model_path} has an unknown ending. Supported suffixes are h5/onnx.")

    return True
