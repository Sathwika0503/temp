// Login validation
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Simple validation (you can replace this with actual authentication)
    if (username === 'user' && password === 'password') {
        window.location.href = 'prediction';
    } else {
        alert('Invalid username or password');
    }
});
