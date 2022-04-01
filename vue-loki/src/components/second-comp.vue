<script>
export default {
  emits: ['response'],

  methods: {
      firethis() {
          this.$emit('response', 'fired!')
      }
  }
}
</script>

<template>
  <p><a href="#" @click.prevent="firethis">2nd component</a></p>
</template>