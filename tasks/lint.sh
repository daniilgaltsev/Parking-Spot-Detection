#!/bin/bash

set -uo pipefail
set +e

for idx in {0..6}
do
    FAILURES[idx]=false
done

path=$(dirname "$0")
cd "$path/.." || { echo "cd failed"; exit 1; }

echo "Running safety check"
safety check -r requirements/requirements.txt -r requirements/requirements-dev.txt --full-report || FAILURES[0]=true
echo "safety failure: ${FAILURES[0]}"

echo "Running pylint"
pylint parking_spot_detection training || FAILURES[1]=true
echo "pylint failure: ${FAILURES[1]}"

echo "Running pycodestyle"
pycodestyle parking_spot_detection training || FAILURES[2]=true
echo "pycodestyle failure: ${FAILURES[2]}"

echo "Running pydocstyle"
pydocstyle parking_spot_detection training || FAILURES[3]=true
echo "pydocstyle failure: ${FAILURES[3]}"

# echo "Running mypy"
# mypy parking_spot_detection training || FAILURES[4]=true
# echo "mypy failure: ${FAILURES[4]}"

echo "Running bandit"
bandit -ll -r parking_spot_detection training || FAILURES[5]=true
echo "bandit failure: ${FAILURES[5]}"

echo "shellcheck"
shellcheck tasks/*.sh || FAILURES[6]=true
echo "shellcheck failure: ${FAILURES[6]}"

for failure in "${FAILURES[@]}"
do
    if [ "$failure" = true ]; then
        echo "Linting failed"
        sleep 30
        exit 1
    fi
done

echo "Linting passed"
sleep 30
exit 0