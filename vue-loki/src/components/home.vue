<template>
  <div class="home">
    <h1>{{ msg }}</h1>
    <p>
      For a reference of Loki's APIs, please visit
      <a href="https://git.local/irish-dev-team/loki" target="_blank" rel="noopener">Loki's official repo</a> or his <a href="/api/fr/docs"  target="_blank">FastAPI's Swagger UI</a>.
    </p>
  </div>
</template>

<script>
export default {
  name: 'home-page',
  props: {
    msg: String
  }
}
</script>