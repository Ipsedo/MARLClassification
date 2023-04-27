

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! [[ -d "${SCRIPT_DIR}/downloaded" ]]; then
    mkdir "${SCRIPT_DIR}/downloaded"
fi

if ! [[ -f "${SCRIPT_DIR}/downloaded/kinetics700_2020.tar.gz" ]]; then
  wget https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020.tar.gz -P "${SCRIPT_DIR}/downloaded"
  tar -xvzf "${SCRIPT_DIR}/downloaded/kinetics700_2020.tar.gz" -C "${SCRIPT_DIR}/downloaded"
fi