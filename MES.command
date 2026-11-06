#!/bin/bash
# Double-click in Finder (macOS): starts MES and opens it in the browser. Closing this window stops MES.
cd "$(dirname "$0")" && exec ./run.sh --open "$@"
