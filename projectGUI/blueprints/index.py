from flask import Blueprint, render_template, session, request
import os
import shutil
import tensorflow as tf

bp = Blueprint("index", __name__, url_prefix="/")


@bp.route('/')
def index():
    tf.keras.backend.clear_session()
    folders_to_clean = ['static/DLFile/output_folder/', 'static/DLFile/upload_folder/', 'static/DLFile/videoPic/outputFrames/', 'static/DLFile/videoPic/uploadFrames/']

    for folder in folders_to_clean:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    return render_template("index.html")
