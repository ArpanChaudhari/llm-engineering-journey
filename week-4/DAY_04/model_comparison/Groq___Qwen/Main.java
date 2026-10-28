public class Main {
    public static void main(String[] args) {
        int iterations = 200000000;
        int param1 = 4;
        int param2 = 1;

        double result = 1.0;
        for (int i = 1; i <= iterations; i++) {
            int j = i * param1 - param2;
            result -= 1.0 / j;
            j = i * param1 + param2;
            result += 1.0 / j;
        }

        result *= 4.0;

        System.out.printf("Result: %.12f%n", result);
    }
}