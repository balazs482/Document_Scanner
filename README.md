# Document_Scanner
Széchenyi István University, Gépi látás (GKNB_INTM038) project


## Goal
The goal is to make a document scanner application.
The application will be using a camera feed to capture an image, find the corners of the sheet, warp it accordingly and improve readability with post-effects.

## Current state
Right now, the program reads the image assigned to the IMAGE_PATH variable in the Document_Scanner.py source code, and if four distinctive edges are recognized, the image warps accordingly and appears in a separate window.