<script>
    export default {

        props: {
            item: Object,
        },

        emits: ['remove-facerep'],

        data() {
            return {

            } 
        },

        methods: {
            populateCanvas(item) {
                var canvas = document.getElementById('picCanvas'+item.id)
                var context = canvas.getContext('2d')
                var imageObj = new Image();
                imageObj.src = '/data/'+item.image_name_orig
                
                imageObj.onload = function() {
                    // console.log(imageObj.width+", "+imageObj.height)
                    var scale = Math.min(canvas.width / imageObj.width, canvas.height / imageObj.height)
                    // get the top left position of the image
                    // var x = (canvas.width / 2) - (imageObj.width / 2) * scale
                    // var y = (canvas.height / 2) - (imageObj.height / 2) * scale
                    context.drawImage(imageObj, 0, 0, imageObj.width * scale, imageObj.height * scale)
                    context.beginPath()
                    // context.moveTo(x, y)
                    context.strokeStyle = "#FF0000"
                    context.lineWidth = 2
                    context.strokeRect(item.region[0]*scale,item.region[1]*scale,item.region[2]*scale,item.region[3]*scale)
                    
                }

            }
        },

        mounted() {
            this.populateCanvas(this.item)
        },
    }
</script>


<template>
    <!-- <img :src="'/data/'+item.image_name_orig" class="img_thumb"/>
    <div>{{ item.region }}</div> -->
    <div class="canvas-container">
        <canvas :id="'picCanvas'+item.id"></canvas>
        <button class='buttCanvas' @click="$emit('remove-facerep', item.id, item.person_id)">X</button>
    </div>

</template>


<style scoped>

.canvas-container {
    position: relative;
}

.buttCanvas {
    position: absolute;
    left: 5px;
    top: 5px;
}

</style>