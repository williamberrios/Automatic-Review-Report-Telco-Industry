<template>
  <q-layout view="lHh Lpr lFf">
    <div class="header-main">
      <q-header elevated>
        <q-toolbar class="bg-secondary text-primary">
          <div class="flex justify-center row" style="position:absolute;left: 0;">
            <div>
              <q-img
                src="images/entel.png"
                spinner-color="white"
                style="width: 80px;"
              />
            </div>
            <div class="title-header">
              Entel - Document data extraction
            </div>
          </div>
          <q-space/>
          <div class="header-actions">
            <router-link :to="{ name: 'home' }">
              <q-btn no-caps flat class="menu-item">
                <q-icon color="white" size="1.1875rem" name="home"></q-icon>
                  Home
              </q-btn>
            </router-link>

            <router-link :to="{ name: 'history' }">
              <q-btn no-caps flat class="menu-item">
                <q-icon color="white" size="1.1875rem" name="timeline"></q-icon>
                  History
              </q-btn>
            </router-link>
          </div>
        </q-toolbar>
      </q-header>
    </div>
    <!-- <q-tabs
      v-model="tab"
      inline-label
      no-caps
      align="right"
      style="position: relative"
      class="no-padding bg-primary text-white"
    >
      <q-img
        src="images/entel.jpeg"
        spinner-color="white"
        style="width: 120px;position:absolute;left: 0;margin: auto 0;"
      />
      <q-route-tab
        v-for="nav in navs"
        :key="nav.label"
        :to="nav.url"
        :icon="nav.icon"
        :label="nav.label"
      >
      </q-route-tab>
    </q-tabs> -->

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script>

export default {
  name: 'MainLayout',
  data () {
    return {
      tab: "",
      navs: [
        {
          label: "Home",
          icon: "home",
          url: "/home"
        },
        {
          label: "Historial",
          icon: "timeline",
          url: "/history"
        }
      ]
    }
  }
}
</script>

<style lang="scss">
//$
.header-main {
  .title-header {
    margin: auto 16px;
    color: white;
    font-size: 22px;
    letter-spacing: 1px;
  }
  .menu-item {
    border-radius: 20px !important;
    color: white;

    transition: all 0.3s;

    .q-btn__wrapper {
      padding: 0;
      min-height: 0;
    }
  }
  .header-actions {
    display: flex;

    //gap: 15px;

    a + * {
      margin-left: 15px;
    }

    .router-link-active {
      .q-btn {
        background: rgba(0, 92, 143, 0.1);
      }
    }
    a {
      color: $primary;
      text-decoration: none;

      &:hover {
          text-decoration: none;
      }

      &.default-color {
          color: $grey;
      }
    }
    .q-btn {
      font-size: 12px;
      line-height: 14px;
      font-weight: 600;
      // @include font(12px, 14px);
      font-family: Raleway;

      padding: 6px 15px;

      &:hover {
        background: rgba(0, 92, 143, 0.1);
      }
    }

    .q-icon {
      margin-right: 10px;
    }
  }
}
</style>
