#!/bin/bash
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NAME=$1

if [[ $NAME != "cartwheel" && $NAME != "reflection" && $NAME != "splash" &&  $NAME != "trampoline" && $NAME != "all" ]]; then
    echo "Available videos are: cartwheel, reflection, splash, trampoline"
    exit 1
fi

if [[ $NAME == "all" ]]; then
  declare -a NAMES=("cartwheel" "reflection" "splash" "trampoline")
else
  declare -a NAMES=($NAME)
fi

for NAME in "${NAMES[@]}"
do
  echo "Specified [$NAME]"
  URL=https://www.robots.ox.ac.uk/~erika/retiming/data/$NAME.zip
  ZIP_FILE=./datasets/$NAME.zip
  TARGET_DIR=./datasets/$NAME/
  wget -N $URL -O $ZIP_FILE
  mkdir $TARGET_DIR
  unzip $ZIP_FILE -d ./datasets/
  rm $ZIP_FILE
done