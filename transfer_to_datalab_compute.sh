#!/bin/bash
# Transfer simulation_data/artifacts/run2000 to datalab using a compute node
# This follows best practices by using a compute node instead of the login node
#
# USAGE:
#   1. Make sure you can SSH from datalab to mitrix (see instructions below)
#   2. SSH to datalab: ssh datalab
#   3. Request an interactive compute node: srun --pty --cpus-per-task=1 --time=4:00:00 bash
#   4. From the compute node, run this script (or copy-paste the rsync command)

# Configuration
MITRIX_USER="root"  # Your username on mitrix
MITRIX_HOST="mitrix"  # Hostname or IP of mitrix (update if needed)
MITRIX_SSH_KEY="$HOME/.ssh/id_ed25519"  # SSH key to access mitrix from datalab
MITRIX_SOURCE="/root/projects/my-herosim/simulation_data/artifacts/run2000"

DATALAB_USER="${USER}"  # Your datalab username (auto-detected)
DEST_PATH="/share/${DATALAB_USER}/simulation_data/artifacts/"

echo "Transferring from mitrix to datalab compute node..."
echo "Source: ${MITRIX_USER}@${MITRIX_HOST}:${MITRIX_SOURCE}"
echo "Destination: ${DEST_PATH}run2000"
echo ""
echo "This may take a while. Press Ctrl+C to cancel."
sleep 3

# Create destination directory
mkdir -p ${DEST_PATH}

# Transfer using rsync (pulling from mitrix)
rsync -avP -e "ssh -i ${MITRIX_SSH_KEY}" \
  ${MITRIX_USER}@${MITRIX_HOST}:${MITRIX_SOURCE} \
  ${DEST_PATH}

echo ""
echo "Transfer complete!"
echo "Data is now at: ${DEST_PATH}run2000"




