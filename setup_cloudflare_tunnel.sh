#!/bin/bash

echo "Installing cloudflared..."
brew install cloudflare/cloudflare/cloudflared

echo "Authenticating with Cloudflare..."
cloudflared tunnel login

echo "Creating tunnel..."
cloudflared tunnel create archimind-mvp

echo "Setting up config file..."
mkdir -p ~/.cloudflared
# Copy config to the standard location
cp cloudflared.yml ~/.cloudflared/config.yml

echo ""
echo "Note: The tunnel credentials file will be created automatically"
echo "in ~/.cloudflared/ when you run 'cloudflared tunnel create'"
echo ""
echo "Done. You can now run the tunnel with:"
echo "cloudflared tunnel run archimind-mvp"

