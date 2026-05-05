// convert_to_avi.ijm
// Converts a single microscopy file to uncompressed grayscale AVI.
//
// Called by convert_to_avi.py via:
//   fiji --headless -macro convert_to_avi.ijm "input_path|output_path|fps"
//
// The input can be any format Bio-Formats supports:
//   .nd2, .tif/.tiff (stack or sequence), .czi, .lif, .isxd, .bmp, .png ...
//
// The output is an uncompressed AVI with the correct FPS embedded in its header.
// IMPORTANT: The pipe | is used as delimiter (not comma) to avoid issues with
// Windows paths that may contain commas.

args = getArgument();
if (args == "") {
    print("ERROR: No arguments provided.");
    print("Usage: fiji --headless -macro convert_to_avi.ijm \"input|output|fps\"");
    exit(1);
}

parts = split(args, "|");
if (parts.length < 3) {
    print("ERROR: Expected 3 arguments separated by |: input|output|fps");
    exit(1);
}

input_path  = parts[0];
output_path = parts[1];
fps         = parseInt(parts[2]);

print("Input:  " + input_path);
print("Output: " + output_path);
print("FPS:    " + fps);

// ── Open the file (Bio-Formats handles all microscopy formats) ────────────────
if (endsWith(input_path, ".tif") || endsWith(input_path, ".tiff")) {
    // For TIFF stacks: open directly
    open(input_path);
} else {
    // For ND2, CZI, LIF, etc.: use Bio-Formats importer
    run("Bio-Formats Importer",
        "open=[" + input_path + "] " +
        "autoscale color_mode=Grayscale rois_import=[ROI manager] " +
        "view=Hyperstack stack_order=XYCZT");
}

if (nImages == 0) {
    print("ERROR: Could not open " + input_path);
    exit(1);
}

title = getTitle();
print("Opened: " + title + "  (" + nSlices + " slices)");

// ── Convert to 8-bit grayscale ────────────────────────────────────────────────
// If already 8-bit, this is a no-op.
// If 16-bit (common in calcium imaging), we scale to 8-bit to keep AVI small.
// NOTE: if you want to keep 16-bit precision, change "8-bit" to "16-bit" below,
// but check that your downstream MPS installation supports it.
if (bitDepth() != 8) {
    print("Converting from " + bitDepth() + "-bit to 8-bit...");
    run("8-bit");
}

// If it's a colour image, convert to grayscale
if (is("composite")) {
    run("Stack to RGB");
    run("8-bit");
}

// ── Set frame rate in image metadata ─────────────────────────────────────────
Stack.setFrameRate(fps);

// ── Save as uncompressed AVI ──────────────────────────────────────────────────
// "compression=None" → raw uncompressed AVI (no codec)
// "frame=fps"       → embed FPS in AVI header
run("AVI... ", "compression=None frame=" + fps + " save=[" + output_path + "]");

print("Saved: " + output_path);
close("*");
print("Done.");
