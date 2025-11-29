<!-- Page2.vue -->
<template>
    <div>
      <div style="text-align: center">
        <runner></runner>
      </div>
      <!-- 页面1的其他内容 -->
    </div>
  </template>
  <script>
  import runner from "@/components/runner/runner";
  export default {
    name: "Page2",
    components: {
        runner
    }
  };
  </script>
