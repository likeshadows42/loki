<script>
export default {
  emits: ['response'],

  methods: {
      firethis() {
        this.$emit('response', 'msg fired!')
      }
  }
}
</script>

<template>
  <p>
    <a href="#" @click.prevent="firethis">test component</a>
  </p>
</template>