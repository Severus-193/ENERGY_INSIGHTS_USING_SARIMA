document.addEventListener('DOMContentLoaded', () => {
    const trackerForm = document.getElementById('trackerForm');
    const output = document.getElementById('output');

    trackerForm.addEventListener('submit', (event) => {
        event.preventDefault();

        const dailyUsage = parseFloat(document.getElementById('dailyUsage').value);
        
        if (isNaN(dailyUsage) || dailyUsage < 0) {
            output.innerHTML = '<p style="color: red;">Please enter a valid number.</p>';
            return;
        }

        // Example calculation
        const monthlyUsage = dailyUsage * 30;
        output.innerHTML = `
            <p>Daily Usage: ${dailyUsage.toFixed(2)} kWh</p>
            <p>Estimated Monthly Usage: ${monthlyUsage.toFixed(2)} kWh</p>
        `;
    });
});
