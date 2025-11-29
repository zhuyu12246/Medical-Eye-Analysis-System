<!-- Page1.vue -->
<template>
    <div>
      <div style="text-align: center">
        <upload></upload>
      </div>
      <div>
        <UserSelect></UserSelect>
      </div>
    </div>
  </template>

  <script>
  import upload from "@/components/upload/upload";
  import UserSelect from "@/components/user-select/user-select";
  export default {
    name: "Page1",
    components: {
        upload,
        UserSelect
    }
  };
  </script>
