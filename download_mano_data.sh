#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

read -p "Username (email):" username
read -p "Password:" -s password

username=$(urle $username)
password=$(urle $password)

mkdir -p manopth/mano/models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip' -O 'manopth/mano/mano.zip' --no-check-certificate --continue
unzip -j "manopth/mano/mano.zip" "mano_v1_2/models/*" -d manopth/mano/models
rm manopth/mano/mano.zip