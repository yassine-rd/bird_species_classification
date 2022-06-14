$(document).ready(function () {

    // Initialisation
    /*
     * Après que la page soit chargée, on cache les éléments ayant comme classe
     * 'image-section' et 'loader', et l'élémnent ayant comme id 'result'
     */

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Overview of the loaded image
    function readURL(input) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
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
                $('#result').fadeIn(1400);
                $('#result').text(' ' + data);
                console.log('Success!');
            },
        });
    });

});
