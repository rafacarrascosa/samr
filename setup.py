import os
from pip.req import parse_requirements

from distutils.core import setup


base_path = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(base_path, "docs", "setup", "requirements.txt")
install_reqs = list(parse_requirements(requirements_path))
reqs = [str(ir.req) for ir in install_reqs]


setup(
    name="kaggle-sentiment-movie-reviews",
    version="0.1",
    description="An entry to kaggle's 'Sentiment Analysis on Movie Reviews' competition",
    author="Rafael Carrascosa",
    packages=["samr"],
    install_requires=reqs,
    scripts=["scripts/generate_kaggle_submission.py",
             "scripts/cross_validate_config.py",
             "scripts/download_3rdparty_data.py"]
)
