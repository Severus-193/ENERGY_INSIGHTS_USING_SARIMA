document.addEventListener('DOMContentLoaded', () => {
    // Select all recommendation buttons
    const recommendButtons = document.querySelectorAll('.recommend-btn');
    
    // Add click event listener to each button
    recommendButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Get the recommendation text from the data attribute
            const recommendationText = this.getAttribute('data-recommendation');
            
            // Find the corresponding text paragraph within the same appliance-card
            const recommendationParagraph = this.nextElementSibling;
            recommendationParagraph.textContent = recommendationText;
        });
    });
});
