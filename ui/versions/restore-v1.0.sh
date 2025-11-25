#!/bin/bash
# Restore UI to Version 1.0 Prototype

echo "🔄 Restoring UI to Version 1.0 Prototype..."

# Restore the template
cp index-v1.0.html ../templates/index.html

# Restore the dashboard
cp dashboard-v1.0.py ../dashboard.py

echo "✅ Version 1.0 restored successfully!"
echo "Restart the Flask server to see the changes."
