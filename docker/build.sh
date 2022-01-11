#!/bin/bash

TARGET='base'
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Usage info
show_help() {
echo "
Usage: ./build.sh [-h]
[--target TARGET]
--target TARGET Multi-stage target in Dockerfile [default=$TARGET]
"
}

while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    -g|--target)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TARGET=$2
            shift
        else
            die 'ERROR: "--target" requires a non-empty option argument.'
        fi
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

docker build \
    -t openpcdet-docker-$TARGET \
    --target $TARGET \
    --build-arg USER_ID=$USER_ID \
    --build-arg GROUP_ID=$GROUP_ID \
    -f Dockerfile \
    ..
