<script>
    export default {

        props: {
            item: Object,
            show_button: Boolean,
        },

        emits: ['parent-handler'],

        data() {
            return {
            } 
        },

        watch: {
            'item.hidden': function() {
                this.populateCanvas(this.item)
            }
        },

        methods: {
            populateCanvas(item) {
                var canvas = document.getElementById('picCanvas'+item.id)
                var context = canvas.getContext('2d')
                context.clearRect(0, 0, canvas.width, canvas.height)
                var imageObj = new Image();
                imageObj.src = '/data/'+item.image_name
                
                imageObj.onload = function() {
                    console.log("image: "+imageObj.width+", "+imageObj.height)
                    console.log("canvas: "+canvas.width+", "+canvas.height)
                    console.log("coords: "+item.region[0]+", "+item.region[1]+", "+item.region[2]+", "+item.region[3])
                    var scale = Math.min(canvas.width / imageObj.width, canvas.height / imageObj.height)
                    // get the top left position of the image
                    // var x = (canvas.width / 2) - (imageObj.width / 2) * scale
                    // var y = (canvas.height / 2) - (imageObj.height / 2) * scale
                    if(item.hidden) {
                        context.globalAlpha = 0.4
                    } else {
                         context.globalAlpha = 1.0
                    }
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
    <div class="canvas-container">
        
        

        <span v-if="show_button">
            <button v-if="item.hidden" class='buttCanvasNoFace' @click="$emit('parent-handler', 'unhide', item.id, item.person_id)">UNHIDE</button>
            <button v-else class='buttCanvasNoFace' @click="$emit('parent-handler', 'hide', item.id, item.person_id)">HIDE</button>
            <button class='buttCanvasRemove' @click="$emit('parent-handler', 'remove', item.id, item.person_id)">REM</button>
        </span>
        <canvas v-if="show_button" :id="'picCanvas'+item.id"></canvas>
        <canvas v-else :id="'picCanvas'+item.id" @click="$emit('parent-handler', item.id, item.person_id)"></canvas>
    </div>
</template>


<style scoped>

.canvas-container {
    position: relative;
}

.buttCanvasNoFace {
    position: absolute;
    left: 5px;
    bottom: 5px;
}

.buttCanvasRemove {
    position: absolute;
    left: 5px;
    bottom: 30px;
}

</style>