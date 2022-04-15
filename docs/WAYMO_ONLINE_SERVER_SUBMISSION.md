# Submit result to waymo evaluation server

### Process test set
~~~
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_test_set_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
~~~

### Generate preds.bin file using result.pkl file for validation set
The scripts assume that you have run data preprocessing script again to generate '../data/waymo/waymo_time_name_infos_val.pkl' to save timestamp and context_name for each frame.
~~~
python tools/create_pred_bin.py result.pkl --save_path pred.bin
~~~

### Generate preds.bin file using ckpt file for test set (assuming test set is already processed)
The scripts assume that you have run data preprocessing script again to generate '../data/waymo/waymo_time_name_infos_test.pkl' to save timestamp and context_name for each frame. It will also save the result.pkl file for test set after running this script.
~~~
python tools/test_set_eval.py --ckpt model.pth --cfg_file tools/cfgs/waymo_models/pv_rcnn.yaml --test_set --save_path pred.bin
~~~

### Generate preds.bin file using result.pkl file for test set
~~~
python tools/create_pred_bin.py result.pkl --testing --save_path pred.bin
~~~
or
~~~
python tools/test_set_eval.py --pickle_file result.pkl --test_set --save_path pred.bin
~~~


### Setup waymo_open_dataset environment

~~~
git clone https://github.com/waymo-research/waymo-open-dataset.git
cd waymo-open-dataset
apt-get update

apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
apt install build-essential
~~~

### Edit WORKSPACE to add the following lines (if the build process fails in later steps):
~~~
http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/archive/262ebec3c2296296526740db4aefce68c80de7fa.tar.gz"],
        strip_prefix = "rules_cc-262ebec3c2296296526740db4aefce68c80de7fa",
        sha256 = "3057c13fa4d431eb0e7a9c28eea13f25987d29f869406b5ee3f2bd9c4134cb0c",
)

http_archive(
    name = "rules_proto",
    sha256 = "2490dca4f249b8a9a3ab07bd1ba6eca085aaf8e45a734af92aad0c42d9dc7aaf",
    strip_prefix = "rules_proto-218ffa7dfa5408492dc86c01ee637614f8695c45",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/218ffa7dfa5408492dc86c01ee637614f8695c45.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/218ffa7dfa5408492dc86c01ee637614f8695c45.tar.gz",
    ],
)

http_archive(
    name = "rules_cc",
    sha256 = "35f2fb4ea0b3e61ad64a369de284e4fbbdcdba71836a5555abb5e194cf119509",
    strip_prefix = "rules_cc-624b5d59dfb45672d4239422fa1e3de1822ee110",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
        "https://github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
    ],
)
~~~

### Sanity Check for bazel build (optional)
If build fails, follow the previous steps to add additional lines into WORKSPACE.
~~~
bazel test waymo_open_dataset/metrics:all

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main waymo_open_dataset/metrics/tools/fake_predictions.bin waymo_open_dataset/metrics/tools/fake_ground_truths.bin
~~~

### Add meta data
Change attributes in waymo_open_dataset/metrics/tools/submission.txtpb to add method names, authors, etc.

### build create_submission and generate submission files
~~~
mkdir /tmp/my_model
bazel build waymo_open_dataset/metrics/tools/create_submission

bazel-bin/waymo_open_dataset/metrics/tools/create_submission --input_filenames='/tmp/preds.bin' --output_filename='/tmp/my_model/model' --submission_filename='waymo_open_dataset/metrics/tools/submission.txtpb'
~~~

### zip the files
~~~
tar cvf /tmp/my_model.tar /tmp/my_model/
gzip /tmp/my_model.tar
~~~

### submit my_model.tar

Submit .gz file to https://waymo.com/open/challenges/2020/3d-detection/#
