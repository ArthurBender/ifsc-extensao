jQuery(function($) {
    function scrollToAnchor(aid) {
        const destination = $("#" + aid);
        $('html,body').animate({
            scrollTop: destination.offset().top
        },'slow');
    }

    $(document).on('click', '.smooth-link', function(){
        console.log("IOPARR");
        scrollToAnchor($(this).attr("link-destination"));
    });
});