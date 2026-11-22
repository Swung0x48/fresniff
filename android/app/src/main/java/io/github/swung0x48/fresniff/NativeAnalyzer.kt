package io.github.swung0x48.fresniff

/** The command line program of fresniff as a native library (libfresniff.so). */
object NativeAnalyzer {
    init {
        System.loadLibrary("fresniff")
    }

    fun interface ProgressListener {
        /** frames read so far and the progress (0 - 1, negative if unknown); false stops the analysis */
        fun onProgress(frames: Long, progress: Double): Boolean
    }

    /**
     * Runs fresniff with these arguments, its messages go to the file logPath.
     * Returns the exit code: 0 success, 1 error, 2 invalid arguments. One call at a time.
     */
    external fun run(arguments: Array<String>, logPath: String, listener: ProgressListener?): Int
}
