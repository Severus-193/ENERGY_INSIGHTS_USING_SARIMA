document.addEventListener('DOMContentLoaded', () => {
    const usageDisplay = document.getElementById('usageDisplay');
    
    // Simulate real-time data
    function fetchRealTimeData() {
        // This should be replaced with actual data fetching logic
        const randomUsage = (Math.random() * 10).toFixed(2);
        usageDisplay.textContent = `${randomUsage} kWh`;
    }
    
    fetchRealTimeData();
    setInterval(fetchRealTimeData, 5000); // Update every 5 seconds

    // Historical data for the chart
    const ctx = document.getElementById('usageChart').getContext('2d');
    const usageChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            datasets: [{
                label: 'Weekly Usage (kWh)',
                data: [12, 19, 3, 5],
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.2)',
                borderWidth: 2,
                pointBackgroundColor: '#007bff',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    beginAtZero: true
                },
                y: {
                    beginAtZero: true
                }
            }
        }
    });
});
