## MLOps
- [![HitCount](http://hits.dwyl.io/zzsza/MLOps.svg)](http://hits.dwyl.io/zzsza/MLOps)
- [머신러닝 오퍼레이션 자동화(MLOps)](https://zzsza.github.io/mlops/2018/12/28/mlops/)와 관련된 내용을 작성하는 Repo입니다

### Model & Pipeline Versioning
* [Data Version Control (DVC)](https://dvc.org/) - A git fork that allows for version management of models
* [ModelDB](https://mitdbg.github.io/modeldb/) - Framework to track all the steps in your ML code to keep track of what version of your model obtained which accuracy, and then visualise it and query it via the UI
* [Pachyderm](https://github.com/pachyderm/pachyderm) - Open source distributed processing framework build on Kubernetes focused mainly on dynamic building of production machine learning pipelines - [(Video)](https://www.youtube.com/watch?v=LamKVhe2RSM)
* [steppy](https://github.com/neptune-ml/steppy) - Lightweight, Python3 library for fast and reproducible machine learning experimentation. Introduces simple interface that enables clean machine learning pipeline design.
* [Jupyter Notebooks](http://jupyter.org/) - Web interface python sandbox environments for reproducible development
* [Quilt Data](https://quiltdata.com/) - Versioning, reproducibility and deployment of data and models.
* [H2O Flow](https://www.h2o.ai/download/) - Jupyter notebook-like inteface for H2O to create, save and re-use "flows"
* [ModelChimp](https://www.modelchimp.com/) - Framework to track and compare all the results and parameters from machine learning models [(Video)](https://vimeo.com/271246650)
* [PredictionIO](https://predictionio.apache.org/) - An open source Machine Learning Server built on top of a state-of-the-art open source stack for developers and data scientists to create predictive engines for any machine learning task
* [MLflow](https://github.com/mlflow/mlflow) - Open source platform to manage the ML lifecycle, including experimentation, reproducibility and deployment.
* [Sacred](https://github.com/IDSIA/sacred) - Tool to help you configure, organize, log and reproduce machine learning experiments.
* [FGLab](https://github.com/Kaixhin/FGLab) - Machine learning dashboard, designed to make prototyping experiments easier.
* [Studio.ML](https://www.studio.ml/) - Model management framework which minimizes the overhead involved with scheduling, running, monitoring and managing artifacts of your machine learning experiments.

### Data Storage / Standardisation / Privacy
* [EdgeDB](https://edgedb.com/) - NoSQL interface for Postgres that allows for object interaction to data stored
* [BayesDB](http://probcomp.csail.mit.edu/bayesdb/) - Database that allows for built-in non-parametric Bayesian model discovery and queryingi for data on a database-like interface - [(Video)](https://www.youtube.com/watch?v=2ws84s6iD1o)
* [Apache Arrow](https://arrow.apache.org/) - In-memory columnar representation of data compatible with Pandas, Hadoop-based systems, etc
* [Apache Parquet](https://parquet.apache.org/) - On-disk columnar representation of data compatible with Pandas, Hadoop-based systems, etc
* [Apache Kafka](https://kafka.apache.org/) - Distributed streaming platform framework
* [Uber SQL Differencial Privacy](https://github.com/uber/sql-differential-privacy) - Uber's open source framework that enforces differential privacy for general-purpose SQL queries.
* [ClickHouse](https://clickhouse.yandex/) - ClickHouse is an open source column oriented database management system supported by Yandex - [(Video)](https://www.youtube.com/watch?v=zbjub8BQPyE)


### Feature Engineering Automation
* [auto-sklearn](https://automl.github.io/auto-sklearn/stable/) - Framework to automate algorithm and hyperparameter tuning for sklearn
* [TPOT](https://epistasislab.github.io/tpot/) - Automation of sklearn pipeline creation (including feature selection, pre-processor, etc)
* [tsfresh](https://github.com/blue-yonder/tsfresh) - Automatic extraction of relevant features from time series
* [Featuretools](https://www.featuretools.com/) - An open source framework for automated feature engineering
* [Colombus](http://i.stanford.edu/hazy/victor/columbus/) - A scalable framework to perform exploratory feature selection implemented in R
* [automl](https://github.com/ClimbsRocks/automl) - Automated feature engineering, feature/model selection, hyperparam. optimisation


### Model Deployment Frameworks
* [Seldon](https://github.com/SeldonIO/seldon-core) - Open source platform for deploying and monitoring machine learning models in kubernetes - [(Video)](https://www.youtube.com/watch?v=pDlapGtecbY)
* [Redis-ML](https://github.com/RedisLabsModules/redis-ml) - Module available from unstable branch that supports a subset of ML models as Redis data types
* [Model Server for Apache MXNet (MMS)](https://github.com/awslabs/mxnet-model-server) - A model server for Apache MXNet from Amazon Web Services that is able to run MXNet models as well as Gluon models (Amazon's SageMaker runs a custom version of MMS under the hood)
* [Tensorflow Serving](https://www.tensorflow.org/serving/) - High-performant framework to serve Tensofrlow models via grpc protocol able to handle 100k requests per second per core
* [Clipper](http://clipper.ai/) - Model server project from Berkeley's Rise Rise Lab which includes a standard RESTful API and supports TensorFlow, Scikit-learn and Caffe models
* [DeepDetect](https://github.com/beniz/deepdetect) - Machine Learning production server for TensorFlow, XGBoost and Cafe models written in C++ and maintained by Jolibrain
* [MLeap](https://github.com/combust/mleap) - Standardisation of pipeline and model serialization for Spark, Tensorflow and sklearn
* [OpenScoring](https://github.com/openscoring/openscoring) - REST web service for scoring PMML models built and maintained by OpenScoring.io
* [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) - Model server created by NVIDIA that runs models in ONNX format, including frameworks such as TensorFlow and MATLAB


### Data Pipeline Frameworks
* [Apache Airflow](https://airflow.apache.org/) - Data Pipeline framework built in Python, including scheduler, DAG definition and a UI for visualisation
* [Luigi](https://github.com/spotify/luigi) - Luigi is a Python module that helps you build complex pipelines of batch jobs, handling dependency resolution, workflow management, visualisation, etc
* [Genie](https://github.com/Netflix/genie) - Job orchestration engine to interface and trigger the execution of jobs from Hadoop-based systems
* [Oozie](http://oozie.apache.org/) - Workflow scheduler for Hadoop jobs


### Infrastructure Orchestration Frameworks
* [Kubeflow](https://github.com/kubeflow/kubeflow) - A cloud native platform for machine learning based on Google’s internal machine learning pipelines.
* [Polyaxon](https://github.com/polyaxon/polyaxon) - A platform for reproducible and scalable machine learning and deep learning on kubernetes. - [(Video)](https://www.youtube.com/watch?v=Iexwrka_hys)

### Optimization of Computation
* [Numba](https://github.com/numba/numba) - A compiler for Python array and numerical functions


### Reference
- [awesome-machine-learning-operations](https://github.com/EthicalML/awesome-machine-learning-operations)
