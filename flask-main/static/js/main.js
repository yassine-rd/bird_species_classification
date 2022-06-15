$(document).ready(function () {

    // Initialisation

    /*
     * After the page is loaded, we hide the elements with class 'image-section' and 'loader'
     * and the element with id 'result
     */

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Overview of the loaded image
    function readURL(input) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')').hide().fadeIn(650);
                // $('#imagePreview').hide();
                // $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('').hide();
        // $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        const form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            url: "/predict",
            type: "POST",
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(1500).text(' ' + data);
                // $('#result').text(' ' + data);
                console.log('Prediction done!');
            },
        });
    });
});
