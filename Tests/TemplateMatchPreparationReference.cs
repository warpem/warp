using System;

namespace Tests;

// Independent CPU oracle for the CUDA extraction tests; not a production fallback.
internal static class TemplateMatchPreparationReference
{
    /// <summary>Remove the observed background before zero-padding an image-boundary patch.</summary>
    public static void CopyCenteredPatch(float[] source, int width, int height, int x, int y, int box, float[] destination)
    {
        Array.Clear(destination);
        int left = Math.Max(0, x), top = Math.Max(0, y);
        int right = Math.Min(width, x + box), bottom = Math.Min(height, y + box);
        if (left >= right || top >= bottom) return;
        double sum = 0;
        for (int row = top; row < bottom; row++)
            for (int col = left; col < right; col++) sum += source[row * width + col];
        float mean = (float)(sum / ((right - left) * (bottom - top)));
        for (int row = top; row < bottom; row++)
            for (int col = left; col < right; col++)
                destination[(row-y)*box+col-x] = source[row*width+col] - mean;
    }
}
