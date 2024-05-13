// JavaScript for collapsing panel
document.getElementById('sequence-panel').addEventListener('mouseenter', function() {
    this.classList.remove('collapsed');
});

document.getElementById('sequence-panel').addEventListener('mouseleave', function() {
    this.classList.add('collapsed');
});
