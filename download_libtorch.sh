#!/bin/bash

URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.4.0%2Bcpu.zip"
EXTERN_DIR="extern"
ZIP_FILE="$EXTERN_DIR/libtorch-2.4.0.zip"

mkdir -p "$EXTERN_DIR"

echo "Downloading libtorch 2.4.0..."
wget -O "$ZIP_FILE" "$URL"

echo "Extracting..."
unzip -q "$ZIP_FILE" -d "$EXTERN_DIR"

echo "Cleaning up zip..."
rm "$ZIP_FILE"

echo "Done! libtorch is at $EXTERN_DIR/libtorch"
