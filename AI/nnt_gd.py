def optimize_single_query(initial_x, learning_rate, num_iterations):

    # f(x) = (x - 2)^2
    def f(x):
        return (x - 2)**2
    
    # Gradient f'(x) = 2 * (x - 2)
    def gradient(x):
        return 2*(x - 2)
    
    # Initialize x
    x = initial_x
    
    for i in range(num_iterations):
        # Single query: get gradient at the current x
        grad = gradient(x)
        
        # Immediately update x in the negative gradient direction
        x = x - learning_rate * grad
        
        # Print progress (optional for debugging)
        print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")
    
    return x

# Example usage
initial_x = 10.0
learning_rate = 0.1
num_iterations = 20

final_x = optimize_single_query(initial_x, learning_rate, num_iterations)
print(f"Final value of x: {final_x:.4f}")