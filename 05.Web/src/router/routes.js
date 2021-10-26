
const routes = [
  {
    path: '/',
    redirect: '/home',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      {
        path: "/home",
        name: "home",
        component: () => import("pages/Home.vue")
      },
      {
        path: "/history",
        name: "history",
        component: () => import("pages/History.vue")
      },
    ]
  },

  // Always leave this as last one,
  // but you can also remove it
  {
    path: '*',
    component: () => import('pages/Error404.vue')
  }
]

export default routes
