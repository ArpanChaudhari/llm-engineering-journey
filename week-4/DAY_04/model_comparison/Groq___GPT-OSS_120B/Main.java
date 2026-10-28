import java.io.PrintWriter;
import java.util.Locale;

public class Main {
    public static void main(String[] args) {
        // Parameters from the original Python program
        final int iterations = 200_000_000;   // number of i‑loops
        final int limit = iterations * 2;    // number of terms in the reduced series

        double sum = 0.0;
        double sign = 1.0; // (+1) for k = 0, then alternates

        // Compute Σ_{k=0}^{2*iterations} (-1)^k / (2k+1)
        // This is mathematically equivalent to the original double‑loop.
        for (int k = 0; k <= limit; k++) {
            double term = sign / (2L * k + 1);
            sum += term;
            sign = -sign; // flip sign for next iteration
        }

        double result = sum * 4.0; // final scaling as in the Python code

        // Print with exactly 12 digits after the decimal point
        PrintWriter out = new PrintWriter(System.out);
        out.printf(Locale.US, "Result: %.12f%n", result);
        out.flush();
    }
}