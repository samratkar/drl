---
layout: default
title: Problems and Flashcards
---

<div class="max-w-4xl mx-auto py-8">
  <div class="flex items-center gap-3 mb-6">
    <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-indigo-600" viewBox="0 0 20 20" fill="currentColor">
      <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
      <path fill-rule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clip-rule="evenodd" />
    </svg>
    <h1 class="text-2xl font-bold text-slate-800 m-0">Problems & Flashcards</h1>
  </div>
  
  <table class="w-full text-left border-collapse table-auto bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
    <thead>
      <tr class="border-b border-slate-200 bg-slate-50">
        <th class="px-4 py-3 text-xs font-bold text-slate-500 uppercase tracking-wider">Webpage</th>
        <th class="px-4 py-3 text-xs font-bold text-slate-500 uppercase tracking-wider w-24 text-center">Action</th>
      </tr>
    </thead>
    <tbody>
      {% for file in site.static_files %}
        {% if file.path contains '/problems/' and file.extname == '.html' %}
          <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
            <td class="px-4 py-3 text-sm font-medium text-slate-700">
              {{ file.name | replace: "-", " " | replace: ".html", "" | capitalize }}
              <div class="text-xs text-slate-400 font-normal mt-0.5">{{ file.name }}</div>
            </td>
            <td class="px-4 py-3 text-center">
              <a href="{{ file.path | relative_url }}" class="inline-flex items-center justify-center px-3 py-1.5 text-xs font-bold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 rounded transition-colors no-underline border border-indigo-100">
                Open
              </a>
            </td>
          </tr>
        {% endif %}
      {% endfor %}
    </tbody>
  </table>
  
  <div class="mt-6">
    <a href="{{ '/' | relative_url }}" class="text-sm font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1 no-underline">
      <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
      </svg>
      Back to Home
    </a>
  </div>
</div>
