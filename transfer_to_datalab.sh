#!/bin/bash
# Transfer simulation_data/artifacts/run2000 to datalab
# 
# USAGE:
#   1. Replace USERNAME with your datalab username
#   2. Make sure your SSH key is at ~/.ssh/id_datalab (or update the path)
#   3. Run: bash transfer_to_datalab.sh

# Configuration - UPDATE THESE
DATALAB_USER="nikola.lukic"  # Replace with your datalab username
SSH_KEY="$HOME/.ssh/nikolalukic167"  # Path to your datalab SSH key
SOURCE_DIR="simulation_data/artifacts/run2000"
DEST_HOST="cluster.datalab.tuwien.ac.at"
DEST_PATH="/share/${DATALAB_USER}/simulation_data/artifacts/"

echo "Transferring ${SOURCE_DIR} (15GB) to datalab..."
echo "Destination: ${DATALAB_USER}@${DEST_HOST}:${DEST_PATH}"
echo ""
echo "This may take a while. Press Ctrl+C to cancel."
sleep 3

# Create destination directory on datalab if it doesn't exist
ssh -i ${SSH_KEY} ${DATALAB_USER}@${DEST_HOST} "mkdir -p ${DEST_PATH}"

# Transfer using rsync
rsync -avP -e "ssh -i ${SSH_KEY}" \
  ${SOURCE_DIR} \
  ${DATALAB_USER}@${DEST_HOST}:${DEST_PATH}

echo ""
echo "Transfer complete!"
echo "Data is now at: ${DEST_PATH}run2000"

